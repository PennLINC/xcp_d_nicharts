FROM pennlinc/xcp_d:unstable

# Install xcp_d extension
COPY . /src/xcp_d_nicharts

ARG VERSION=0.0.1

# Force static versioning within container
RUN echo "${VERSION}" > /src/xcp_d_nicharts/xcp_d_nicharts/VERSION && \
    echo "include xcp_d_nicharts/VERSION" >> /src/xcp_d_nicharts/MANIFEST.in && \
    pip install --no-cache-dir "/src/xcp_d_nicharts[all]"

RUN find $HOME -type d -exec chmod go=u {} + && \
    find $HOME -type f -exec chmod go=u {} + && \
    rm -rf $HOME/.npm $HOME/.conda $HOME/.empty

RUN ldconfig
WORKDIR /tmp/

ENTRYPOINT ["/usr/local/miniconda/bin/xcp_d_ukb"]

ARG BUILD_DATE
ARG VCS_REF
ARG VERSION
LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="xcp_d_nicharts" \
      org.label-schema.description="xcp_d_nicharts- postprocessing of UK Biobank outputs" \
      org.label-schema.url="https://xcp-d.readthedocs.io/" \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/PennLINC/xcp_d_nicharts" \
      org.label-schema.version=$VERSION \
      org.label-schema.schema-version="1.0"
